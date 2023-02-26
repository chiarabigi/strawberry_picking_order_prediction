import cv2

def get_segmentation_annotations(segmentation_mask, DEBUG=True):
    hw = segmentation_mask.shape[:2]
    segmentation_mask = segmentation_mask.reshape(hw)
    polygons = []

    for segtype in SegmentationClass.values():
        seg_type_name = proto_api.get_segmentation_type_name(segtype)
        if segtype == SegmentationClass.BACKGROUND:
            continue
        temp_img = np.zeros(hw)
        seg_class_mask_over_seg_img = np.where(segmentation_mask==segtype)
        if np.any(seg_class_mask_over_seg_img):
            temp_img[seg_class_mask_over_seg_img] = 1
            contours, _ = cv2.findContours(temp_img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if len(contours) < 1:
                continue
            has_relevant_contour = False
            for contour in contours:
                if cv2.contourArea(contour) < 2:
                    continue
                has_relevant_contour = True
                polygons.append((contour, seg_type_name))
            if DEBUG and has_relevant_contour:
                fig = plt.figure()
                fig.suptitle(seg_type_name)
                plt.imshow(temp_img)
    return polygons

def get_segmentation_dict(segmentation_mask, img_id="0", starting_annotation_indx=0, DEBUG=False):
    annotations = []
    for indx, (contour, seg_type) in enumerate(get_segmentation_annotations(segmentation_mask, DEBUG=DEBUG)):
        segmentation = contour.ravel().tolist()
        annotations.append({
            "segmentation": segmentation,
            "area": cv2.contourArea(contour),
            "image_id": img_id,
            "category_id": seg_type,
            "id": starting_annotation_indx + indx
        })
    return annotations